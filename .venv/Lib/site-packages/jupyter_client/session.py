"""Session object for building, serializing, sending, and receiving messages.

The Session object supports serialization, HMAC signatures,
and metadata on messages.

Also defined here are utilities for working with Sessions:
* A SessionFactory to be used as a base class for configurables that work with
Sessions.
* A Message object for convenience that allows attribute-access to the msg dict.
"""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import hashlib
import hmac
import json
import logging
import os
import pickle
import pprint
import random
import typing as t
import warnings
from binascii import b2a_hex
from datetime import datetime, timezone
from hmac import compare_digest

# We are using compare_digest to limit the surface of timing attacks
from typing import Optional, Union

import zmq.asyncio
from tornado.ioloop import IOLoop
from traitlets import (
    Any,
    Bool,
    CBytes,
    CUnicode,
    Dict,
    DottedObjectName,
    Instance,
    Integer,
    Set,
    TraitError,
    Unicode,
    observe,
)
from traitlets.config.configurable import Configurable, LoggingConfigurable
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream

from ._version import protocol_version
from .adapter import adapt
from .jsonutil import extract_dates, json_clean, json_default, squash_dates

PICKLE_PROTOCOL = pickle.DEFAULT_PROTOCOL

utc = timezone.utc

# -----------------------------------------------------------------------------
# utility functions
# -----------------------------------------------------------------------------


def squash_unicode(obj):
    """coerce unicode back to bytestrings."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            obj[key] = squash_unicode(obj[key])
            if isinstance(key, str):
                obj[squash_unicode(key)] = obj.pop(key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = squash_unicode(v)
    elif isinstance(obj, str):
        obj = obj.encode("utf8")
    return obj


# -----------------------------------------------------------------------------
# globals and defaults
# -----------------------------------------------------------------------------

# default values for the thresholds:
MAX_ITEMS = 64
MAX_BYTES = 1024

# ISO8601-ify datetime objects
# allow unicode
# disallow nan, because it's not actually valid JSON


def json_packer(obj):
    """Convert a json object to a bytes."""
    try:
        return json.dumps(
            obj,
            default=json_default,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8", errors="surrogateescape")
    except (TypeError, ValueError) as e:
        # Fallback to trying to clean the json before serializing
        packed = json.dumps(
            json_clean(obj),
            default=json_default,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8", errors="surrogateescape")

        warnings.warn(
            f"Message serialization failed with:\n{e}\n"
            "Supporting this message is deprecated in jupyter-client 7, please make "
            "sure your message is JSON-compliant",
            stacklevel=2,
        )

        return packed


def json_unpacker(s):
    """Convert a json bytes or string to an object."""
    if isinstance(s, bytes):
        s = s.decode("utf8", "replace")
    return json.loads(s)


def pickle_packer(o):
    """Pack an object using the pickle module."""
    return pickle.dumps(squash_dates(o), PICKLE_PROTOCOL)


pickle_unpacker = pickle.loads

default_packer = json_packer
default_unpacker = json_unpacker

DELIM = b"<IDS|MSG>"
# singleton dummy tracker, which will always report as done
DONE = zmq.MessageTracker()

# -----------------------------------------------------------------------------
# Mixin tools for apps that use Sessions
# -----------------------------------------------------------------------------


def new_id() -> str:
    """Generate a new random id.

    Avoids problematic runtime import in stdlib uuid on Python 2.

    Returns
    -------

    id string (16 random bytes as hex-encoded text, chunks separated by '-')
    """
    buf = os.urandom(16)
    return "-".join(b2a_hex(x).decode("ascii") for x in (buf[:4], buf[4:]))


def new_id_bytes() -> bytes:
    """Return new_id as ascii bytes"""
    return new_id().encode("ascii")


session_aliases = {
    "ident": "Session.session",
    "user": "Session.username",
    "keyfile": "Session.keyfile",
}

session_flags = {
    "secure": (
        {"Session": {"key": new_id_bytes(), "keyfile": ""}},
        """Use HMAC digests for authentication of messages.
        Setting this flag will generate a new UUID to use as the HMAC key.
        """,
    ),
    "no-secure": (
        {"Session": {"key": b"", "keyfile": ""}},
        """Don't authenticate messages.""",
    ),
}


def default_secure(cfg: t.Any) -> None:  # pragma: no cover
    """Set the default behavior for a config environment to be secure.

    If Session.key/keyfile have not been set, set Session.key to
    a new random UUID.
    """
    warnings.warn("default_secure is deprecated", DeprecationWarning, stacklevel=2)
    if "Session" in cfg and ("key" in cfg.Session or "keyfile" in cfg.Session):
        return
    # key/keyfile not specified, generate new UUID:
    cfg.Session.key = new_id_bytes()


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp"""
    return datetime.utcnow().replace(tzinfo=utc)  # noqa


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------


class SessionFactory(LoggingConfigurable):
    """The Base class for configurables that have a Session, Context, logger,
    and IOLoop.
    """

    logname = Unicode("")

    @observe("logname")  # type:ignore[misc]
    def _logname_changed(self, change: t.Any) -> None:
        self.log = logging.getLogger(change["new"])

    # not configurable:
    context = Instance("zmq.Context")

    def _context_default(self) -> zmq.Context:
        return zmq.Context()

    session = Instance("jupyter_client.session.Session", allow_none=True)

    loop = Instance("tornado.ioloop.IOLoop")

    def _loop_default(self):
        return IOLoop.current()

    def __init__(self, **kwargs):
        """Initialize a session factory."""
        super().__init__(**kwargs)

        if self.session is None:
            # construct the session
            self.session = Session(**kwargs)


class Message:
    """A simple message object that maps dict keys to attributes.

    A Message can be created from a dict and a dict from a Message instance
    simply by calling dict(msg_obj)."""

    def __init__(self, msg_dict: t.Dict[str, t.Any]) -> None:
        """Initialize a message."""
        dct = self.__dict__
        for k, v in dict(msg_dict).items():
            if isinstance(v, dict):
                v = Message(v)  # noqa
            dct[k] = v

    # Having this iterator lets dict(msg_obj) work out of the box.
    def __iter__(self) -> t.ItemsView[str, t.Any]:
        return iter(self.__dict__.items())  # type:ignore[return-value]

    def __repr__(self) -> str:
        return repr(self.__dict__)

    def __str__(self) -> str:
        return pprint.pformat(self.__dict__)

    def __contains__(self, k: object) -> bool:
        return k in self.__dict__

    def __getitem__(self, k: str) -> t.Any:
        return self.__dict__[k]


def msg_header(msg_id: str, msg_type: str, username: str, session: "Session") -> t.Dict[str, t.Any]:
    """Create a new message header"""
    date = utcnow()
    version = protocol_version
    return locals()


def extract_header(msg_or_header: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    """Given a message or header, return the header."""
    if not msg_or_header:
        return {}
    try:
        # See if msg_or_header is the entire message.
        h = msg_or_header["header"]
    except KeyError:
        try:
            # See if msg_or_header is just the header
            h = msg_or_header["msg_id"]
        except KeyError:
            raise
        else:
            h = msg_or_header
    if not isinstance(h, dict):
        h = dict(h)
    return h


class Session(Configurable):
    """Object for handling serialization and sending of messages.

    The Session object handles building messages and sending them
    with ZMQ sockets or ZMQStream objects.  Objects can communicate with each
    other over the network via Session objects, and only need to work with the
    dict-based IPython message spec. The Session will handle
    serialization/deserialization, security, and metadata.

    Sessions support configurable serialization via packer/unpacker traits,
    and signing with HMAC digests via the key/keyfile traits.

    Parameters
    ----------

    debug : bool
        whether to trigger extra debugging statements
    packer/unpacker : str : 'json', 'pickle' or import_string
        importstrings for methods to serialize message parts.  If just
        'json' or 'pickle', predefined JSON and pickle packers will be used.
        Otherwise, the entire importstring must be used.

        The functions must accept at least valid JSON input, and output *bytes*.

        For example, to use msgpack:
        packer = 'msgpack.packb', unpacker='msgpack.unpackb'
    pack/unpack : callables
        You can also set the pack/unpack callables for serialization directly.
    session : bytes
        the ID of this Session object.  The default is to generate a new UUID.
    username : unicode
        username added to message headers.  The default is to ask the OS.
    key : bytes
        The key used to initialize an HMAC signature.  If unset, messages
        will not be signed or checked.
    keyfile : filepath
        The file containing a key.  If this is set, `key` will be initialized
        to the contents of the file.

    """

    debug = Bool(False, config=True, help="""Debug output in the Session""")

    check_pid = Bool(
        True,
        config=True,
        help="""Whether to check PID to protect against calls after fork.

        This check can be disabled if fork-safety is handled elsewhere.
        """,
    )

    packer = DottedObjectName(
        "json",
        config=True,
        help="""The name of the packer for serializing messages.
            Should be one of 'json', 'pickle', or an import name
            for a custom callable serializer.""",
    )

    @observe("packer")
    def _packer_changed(self, change):
        new = change["new"]
        if new.lower() == "json":
            self.pack = json_packer
            self.unpack = json_unpacker
            self.unpacker = new
        elif new.lower() == "pickle":
            self.pack = pickle_packer
            self.unpack = pickle_unpacker
            self.unpacker = new
        else:
            self.pack = import_item(str(new))

    unpacker = DottedObjectName(
        "json",
        config=True,
        help="""The name of the unpacker for unserializing messages.
        Only used with custom functions for `packer`.""",
    )

    @observe("unpacker")
    def _unpacker_changed(self, change):
        new = change["new"]
        if new.lower() == "json":
            self.pack = json_packer
            self.unpack = json_unpacker
            self.packer = new
        elif new.lower() == "pickle":
            self.pack = pickle_packer
            self.unpack = pickle_unpacker
            self.packer = new
        else:
            self.unpack = import_item(str(new))

    session = CUnicode("", config=True, help="""The UUID identifying this session.""")

    def _session_default(self) -> str:
        u = new_id()
        self.bsession = u.encode("ascii")
        return u

    @observe("session")
    def _session_changed(self, change):
        self.bsession = self.session.encode("ascii")

    # bsession is the session as bytes
    bsession = CBytes(b"")

    username = Unicode(
        os.environ.get("USER", "username"),
        help="""Username for the Session. Default is your system username.""",
        config=True,
    )

    metadata = Dict(
        {},
        config=True,
        help="Metadata dictionary, which serves as the default top-level metadata dict for each "
        "message.",
    )

    # if 0, no adapting to do.
    adapt_version = Integer(0)

    # message signature related traits:

    key = CBytes(config=True, help="""execution key, for signing messages.""")

    def _key_default(self) -> bytes:
        return new_id_bytes()

    @observe("key")
    def _key_changed(self, change):
        self._new_auth()

    signature_scheme = Unicode(
        "hmac-sha256",
        config=True,
        help="""The digest scheme used to construct the message signatures.
        Must have the form 'hmac-HASH'.""",
    )

    @observe("signature_scheme")
    def _signature_scheme_changed(self, change):
        new = change["new"]
        if not new.startswith("hmac-"):
            raise TraitError("signature_scheme must start with 'hmac-', got %r" % new)
        hash_name = new.split("-", 1)[1]
        try:
            self.digest_mod = getattr(hashlib, hash_name)
        except AttributeError as e:
            raise TraitError("hashlib has no such attribute: %s" % hash_name) from e
        self._new_auth()

    digest_mod = Any()

    def _digest_mod_default(self) -> t.Callable:
        return hashlib.sha256

    auth = Instance(hmac.HMAC, allow_none=True)

    def _new_auth(self) -> None:
        if self.key:
            self.auth = hmac.HMAC(self.key, digestmod=self.digest_mod)
        else:
            self.auth = None

    digest_history = Set()
    digest_history_size = Integer(
        2**16,
        config=True,
        help="""The maximum number of digests to remember.

        The digest history will be culled when it exceeds this value.
        """,
    )

    keyfile = Unicode("", config=True, help="""path to file containing execution key.""")

    @observe("keyfile")
    def _keyfile_changed(self, change):
        with open(change["new"], "rb") as f:
            self.key = f.read().strip()

    # for protecting against sends from forks
    pid = Integer()

    # serialization traits:

    pack = Any(default_packer)  # the actual packer function

    @observe("pack")
    def _pack_changed(self, change):
        new = change["new"]
        if not callable(new):
            raise TypeError("packer must be callable, not %s" % type(new))

    unpack = Any(default_unpacker)  # the actual packer function

    @observe("unpack")
    def _unpack_changed(self, change):
        # unpacker is not checked - it is assumed to be
        new = change["new"]
        if not callable(new):
            raise TypeError("unpacker must be callable, not %s" % type(new))

    # thresholds:
    copy_threshold = Integer(
        2**16,
        config=True,
        help="Threshold (in bytes) beyond which a buffer should be sent without copying.",
    )
    buffer_threshold = Integer(
        MAX_BYTES,
        config=True,
        help="Threshold (in bytes) beyond which an object's buffer should be extracted to avoid "
        "pickling.",
    )
    item_threshold = Integer(
        MAX_ITEMS,
        config=True,
        help="""The maximum number of items for a container to be introspected for custom serialization.
        Containers larger than this are pickled outright.
        """,
    )

    def __init__(self, **kwargs):
        """create a Session object

        Parameters
        ----------

        debug : bool
            whether to trigger extra debugging statements
        packer/unpacker : str : 'json', 'pickle' or import_string
            importstrings for methods to serialize message parts.  If just
            'json' or 'pickle', predefined JSON and pickle packers will be used.
            Otherwise, the entire importstring must be used.

            The functions must accept at least valid JSON input, and output
            *bytes*.

            For example, to use msgpack:
            packer = 'msgpack.packb', unpacker='msgpack.unpackb'
        pack/unpack : callables
            You can also set the pack/unpack callables for serialization
            directly.
        session : unicode (must be ascii)
            the ID of this Session object.  The default is to generate a new
            UUID.
        bsession : bytes
            The session as bytes
        username : unicode
            username added to message headers.  The default is to ask the OS.
        key : bytes
            The key used to initialize an HMAC signature.  If unset, messages
            will not be signed or checked.
        signature_scheme : str
            The message digest scheme. Currently must be of the form 'hmac-HASH',
            where 'HASH' is a hashing function available in Python's hashlib.
            The default is 'hmac-sha256'.
            This is ignored if 'key' is empty.
        keyfile : filepath
            The file containing a key.  If this is set, `key` will be
            initialized to the contents of the file.
        """
        super().__init__(**kwargs)
        self._check_packers()
        self.none = self.pack({})
        # ensure self._session_default() if necessary, so bsession is defined:
        self.session  # noqa
        self.pid = os.getpid()
        self._new_auth()
        if not self.key:
            get_logger().warning(
                "Message signing is disabled.  This is insecure and not recommended!"
            )

    def clone(self) -> "Session":
        """Create a copy of this Session

        Useful when connecting multiple times to a given kernel.
        This prevents a shared digest_history warning about duplicate digests
        due to multiple connections to IOPub in the same process.

        .. versionadded:: 5.1
        """
        # make a copy
        new_session = type(self)()
        for name in self.traits():
            setattr(new_session, name, getattr(self, name))
        # fork digest_history
        new_session.digest_history = set()
        new_session.digest_history.update(self.digest_history)
        return new_session

    message_count = 0

    @property
    def msg_id(self) -> str:
        message_number = self.message_count
        self.message_count += 1
        return f"{self.session}_{os.getpid()}_{message_number}"

    def _check_packers(self) -> None:
        """check packers for datetime support."""
        pack = self.pack
        unpack = self.unpack

        # check simple serialization
        msg_list = {"a": [1, "hi"]}
        try:
            packed = pack(msg_list)
        except Exception as e:
            msg = f"packer '{self.packer}' could not serialize a simple message: {e}"
            raise ValueError(msg) from e

        # ensure packed message is bytes
        if not isinstance(packed, bytes):
            raise ValueError("message packed to %r, but bytes are required" % type(packed))

        # check that unpack is pack's inverse
        try:
            unpacked = unpack(packed)
            assert unpacked == msg_list
        except Exception as e:
            msg = (
                f"unpacker '{self.unpacker}' could not handle output from packer"
                f" '{self.packer}': {e}"
            )
            raise ValueError(msg) from e

        # check datetime support
        msg_datetime = {"t": utcnow()}
        try:
            unpacked = unpack(pack(msg_datetime))
            if isinstance(unpacked["t"], datetime):
                msg = "Shouldn't deserialize to datetime"
                raise ValueError(msg)
        except Exception:
            self.pack = lambda o: pack(squash_dates(o))
            self.unpack = lambda s: unpack(s)

    def msg_header(self, msg_type: str) -> t.Dict[str, t.Any]:
        """Create a header for a message type."""
        return msg_header(self.msg_id, msg_type, self.username, self.session)

    def msg(
        self,
        msg_type: str,
        content: t.Optional[t.Dict] = None,
        parent: t.Optional[t.Dict[str, t.Any]] = None,
        header: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> t.Dict[str, t.Any]:
        """Return the nested message dict.

        This format is different from what is sent over the wire. The
        serialize/deserialize methods converts this nested message dict to the wire
        format, which is a list of message parts.
        """
        msg = {}
        header = self.msg_header(msg_type) if header is None else header
        msg["header"] = header
        msg["msg_id"] = header["msg_id"]
        msg["msg_type"] = header["msg_type"]
        msg["parent_header"] = {} if parent is None else extract_header(parent)
        msg["content"] = {} if content is None else content
        msg["metadata"] = self.metadata.copy()
        if metadata is not None:
            msg["metadata"].update(metadata)
        return msg

    def sign(self, msg_list: t.List) -> bytes:
        """Sign a message with HMAC digest. If no auth, return b''.

        Parameters
        ----------
        msg_list : list
            The [p_header,p_parent,p_content] part of the message list.
        """
        if self.auth is None:
            return b""
        h = self.auth.copy()
        for m in msg_list:
            h.update(m)
        return h.hexdigest().encode()

    def serialize(
        self,
        msg: t.Dict[str, t.Any],
        ident: t.Optional[t.Union[t.List[bytes], bytes]] = None,
    ) -> t.List[bytes]:
        """Serialize the message components to bytes.

        This is roughly the inverse of deserialize. The serialize/deserialize
        methods work with full message lists, whereas pack/unpack work with
        the individual message parts in the message list.

        Parameters
        ----------
        msg : dict or Message
            The next message dict as returned by the self.msg method.

        Returns
        -------
        msg_list : list
            The list of bytes objects to be sent with the format::

                [ident1, ident2, ..., DELIM, HMAC, p_header, p_parent,
                 p_metadata, p_content, buffer1, buffer2, ...]

            In this list, the ``p_*`` entities are the packed or serialized
            versions, so if JSON is used, these are utf8 encoded JSON strings.
        """
        content = msg.get("content", {})
        if content is None:
            content = self.none
        elif isinstance(content, dict):
            content = self.pack(content)
        elif isinstance(content, bytes):
            # content is already packed, as in a relayed message
            pass
        elif isinstance(content, str):
            # should be bytes, but JSON often spits out unicode
            content = content.encode("utf8")
        else:
            raise TypeError("Content incorrect type: %s" % type(content))

        real_message = [
            self.pack(msg["header"]),
            self.pack(msg["parent_header"]),
            self.pack(msg["metadata"]),
            content,
        ]

        to_send = []

        if isinstance(ident, list):
            # accept list of idents
            to_send.extend(ident)
        elif ident is not None:
            to_send.append(ident)
        to_send.append(DELIM)

        signature = self.sign(real_message)
        to_send.append(signature)

        to_send.extend(real_message)

        return to_send

    def send(
        self,
        stream: Optional[Union[zmq.sugar.socket.Socket, ZMQStream]],
        msg_or_type: t.Union[t.Dict[str, t.Any], str],
        content: t.Optional[t.Dict[str, t.Any]] = None,
        parent: t.Optional[t.Dict[str, t.Any]] = None,
        ident: t.Optional[t.Union[bytes, t.List[bytes]]] = None,
        buffers: t.Optional[t.List[bytes]] = None,
        track: bool = False,
        header: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> t.Optional[t.Dict[str, t.Any]]:
        """Build and send a message via stream or socket.

        The message format used by this function internally is as follows:

        [ident1,ident2,...,DELIM,HMAC,p_header,p_parent,p_content,
         buffer1,buffer2,...]

        The serialize/deserialize methods convert the nested message dict into this
        format.

        Parameters
        ----------

        stream : zmq.Socket or ZMQStream
            The socket-like object used to send the data.
        msg_or_type : str or Message/dict
            Normally, msg_or_type will be a msg_type unless a message is being
            sent more than once. If a header is supplied, this can be set to
            None and the msg_type will be pulled from the header.

        content : dict or None
            The content of the message (ignored if msg_or_type is a message).
        header : dict or None
            The header dict for the message (ignored if msg_to_type is a message).
        parent : Message or dict or None
            The parent or parent header describing the parent of this message
            (ignored if msg_or_type is a message).
        ident : bytes or list of bytes
            The zmq.IDENTITY routing path.
        metadata : dict or None
            The metadata describing the message
        buffers : list or None
            The already-serialized buffers to be appended to the message.
        track : bool
            Whether to track.  Only for use with Sockets, because ZMQStream
            objects cannot track messages.


        Returns
        -------
        msg : dict
            The constructed message.
        """
        if not isinstance(stream, zmq.Socket):
            # ZMQStreams and dummy sockets do not support tracking.
            track = False

        if isinstance(stream, zmq.asyncio.Socket):
            assert stream is not None
            stream = zmq.Socket.shadow(stream.underlying)

        if isinstance(msg_or_type, (Message, dict)):
            # We got a Message or message dict, not a msg_type so don't
            # build a new Message.
            msg = msg_or_type
            buffers = buffers or msg.get("buffers", [])
        else:
            msg = self.msg(
                msg_or_type,
                content=content,
                parent=parent,
                header=header,
                metadata=metadata,
            )
        if self.check_pid and os.getpid() != self.pid:
            get_logger().warning("WARNING: attempted to send message from fork\n%s", msg)
            return None
        buffers = [] if buffers is None else buffers
        for idx, buf in enumerate(buffers):
            if isinstance(buf, memoryview):
                view = buf
            else:
                try:
                    # check to see if buf supports the buffer protocol.
                    view = memoryview(buf)
                except TypeError as e:
                    emsg = "Buffer objects must support the buffer protocol."
                    raise TypeError(emsg) from e
            # memoryview.contiguous is new in 3.3,
            # just skip the check on Python 2
            if hasattr(view, "contiguous") and not view.contiguous:
                # zmq requires memoryviews to be contiguous
                raise ValueError("Buffer %i (%r) is not contiguous" % (idx, buf))

        if self.adapt_version:
            msg = adapt(msg, self.adapt_version)
        to_send = self.serialize(msg, ident)
        to_send.extend(buffers)
        longest = max([len(s) for s in to_send])
        copy = longest < self.copy_threshold

        if stream and buffers and track and not copy:
            # only really track when we are doing zero-copy buffers
            tracker = stream.send_multipart(to_send, copy=False, track=True)
        elif stream:
            # use dummy tracker, which will be done immediately
            tracker = DONE
            stream.send_multipart(to_send, copy=copy)

        if self.debug:
            pprint.pprint(msg)  # noqa
            pprint.pprint(to_send)  # noqa
            pprint.pprint(buffers)  # noqa

        msg["tracker"] = tracker

        return msg

    def send_raw(
        self,
        stream: zmq.sugar.socket.Socket,
        msg_list: t.List,
        flags: int = 0,
        copy: bool = True,
        ident: t.Optional[t.Union[bytes, t.List[bytes]]] = None,
    ) -> None:
        """Send a raw message via ident path.

        This method is used to send a already serialized message.

        Parameters
        ----------
        stream : ZMQStream or Socket
            The ZMQ stream or socket to use for sending the message.
        msg_list : list
            The serialized list of messages to send. This only includes the
            [p_header,p_parent,p_metadata,p_content,buffer1,buffer2,...] portion of
            the message.
        ident : ident or list
            A single ident or a list of idents to use in sending.
        """
        to_send = []
        if isinstance(ident, bytes):
            ident = [ident]
        if ident is not None:
            to_send.extend(ident)

        to_send.append(DELIM)
        # Don't include buffers in signature (per spec).
        to_send.append(self.sign(msg_list[0:4]))
        to_send.extend(msg_list)
        if isinstance(stream, zmq.asyncio.Socket):
            stream = zmq.Socket.shadow(stream.underlying)
        stream.send_multipart(to_send, flags, copy=copy)

    def recv(
        self,
        socket: zmq.sugar.socket.Socket,
        mode: int = zmq.NOBLOCK,
        content: bool = True,
        copy: bool = True,
    ) -> t.Tuple[t.Optional[t.List[bytes]], t.Optional[t.Dict[str, t.Any]]]:
        """Receive and unpack a message.

        Parameters
        ----------
        socket : ZMQStream or Socket
            The socket or stream to use in receiving.

        Returns
        -------
        [idents], msg
            [idents] is a list of idents and msg is a nested message dict of
            same format as self.msg returns.
        """
        if isinstance(socket, ZMQStream):
            socket = socket.socket
        if isinstance(socket, zmq.asyncio.Socket):
            socket = zmq.Socket.shadow(socket.underlying)

        try:
            msg_list = socket.recv_multipart(mode, copy=copy)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                # We can convert EAGAIN to None as we know in this case
                # recv_multipart won't return None.
                return None, None
            else:
                raise
        # split multipart message into identity list and message dict
        # invalid large messages can cause very expensive string comparisons
        idents, msg_list = self.feed_identities(msg_list, copy)
        try:
            return idents, self.deserialize(msg_list, content=content, copy=copy)
        except Exception as e:
            # TODO: handle it
            raise e

    def feed_identities(
        self, msg_list: t.Union[t.List[bytes], t.List[zmq.Message]], copy: bool = True
    ) -> t.Tuple[t.List[bytes], t.Union[t.List[bytes], t.List[zmq.Message]]]:
        """Split the identities from the rest of the message.

        Feed until DELIM is reached, then return the prefix as idents and
        remainder as msg_list. This is easily broken by setting an IDENT to DELIM,
        but that would be silly.

        Parameters
        ----------
        msg_list : a list of Message or bytes objects
            The message to be split.
        copy : bool
            flag determining whether the arguments are bytes or Messages

        Returns
        -------
        (idents, msg_list) : two lists
            idents will always be a list of bytes, each of which is a ZMQ
            identity. msg_list will be a list of bytes or zmq.Messages of the
            form [HMAC,p_header,p_parent,p_content,buffer1,buffer2,...] and
            should be unpackable/unserializable via self.deserialize at this
            point.
        """
        if copy:
            msg_list = t.cast(t.List[bytes], msg_list)
            idx = msg_list.index(DELIM)
            return msg_list[:idx], msg_list[idx + 1 :]
        else:
            msg_list = t.cast(t.List[zmq.Message], msg_list)
            failed = True
            for idx, m in enumerate(msg_list):  # noqa
                if m.bytes == DELIM:
                    failed = False
                    break
            if failed:
                msg = "DELIM not in msg_list"
                raise ValueError(msg)
            idents, msg_list = msg_list[:idx], msg_list[idx + 1 :]
            return [bytes(m.bytes) for m in idents], msg_list

    def _add_digest(self, signature: bytes) -> None:
        """add a digest to history to protect against replay attacks"""
        if self.digest_history_size == 0:
            # no history, never add digests
            return

        self.digest_history.add(signature)
        if len(self.digest_history) > self.digest_history_size:
            # threshold reached, cull 10%
            self._cull_digest_history()

    def _cull_digest_history(self) -> None:
        """cull the digest history

        Removes a randomly selected 10% of the digest history
        """
        current = len(self.digest_history)
        n_to_cull = max(int(current // 10), current - self.digest_history_size)
        if n_to_cull >= current:
            self.digest_history = set()
            return
        to_cull = random.sample(tuple(sorted(self.digest_history)), n_to_cull)
        self.digest_history.difference_update(to_cull)

    def deserialize(
        self,
        msg_list: t.Union[t.List[bytes], t.List[zmq.Message]],
        content: bool = True,
        copy: bool = True,
    ) -> t.Dict[str, t.Any]:
        """Unserialize a msg_list to a nested message dict.

        This is roughly the inverse of serialize. The serialize/deserialize
        methods work with full message lists, whereas pack/unpack work with
        the individual message parts in the message list.

        Parameters
        ----------
        msg_list : list of bytes or Message objects
            The list of message parts of the form [HMAC,p_header,p_parent,
            p_metadata,p_content,buffer1,buffer2,...].
        content : bool (True)
            Whether to unpack the content dict (True), or leave it packed
            (False).
        copy : bool (True)
            Whether msg_list contains bytes (True) or the non-copying Message
            objects in each place (False).

        Returns
        -------
        msg : dict
            The nested message dict with top-level keys [header, parent_header,
            content, buffers].  The buffers are returned as memoryviews.
        """
        minlen = 5
        message = {}
        if not copy:
            # pyzmq didn't copy the first parts of the message, so we'll do it
            msg_list = t.cast(t.List[zmq.Message], msg_list)
            msg_list_beginning = [bytes(msg.bytes) for msg in msg_list[:minlen]]
            msg_list = t.cast(t.List[bytes], msg_list)
            msg_list = msg_list_beginning + msg_list[minlen:]
        msg_list = t.cast(t.List[bytes], msg_list)
        if self.auth is not None:
            signature = msg_list[0]
            if not signature:
                msg = "Unsigned Message"
                raise ValueError(msg)
            if signature in self.digest_history:
                raise ValueError("Duplicate Signature: %r" % signature)
            if content:
                # Only store signature if we are unpacking content, don't store if just peeking.
                self._add_digest(signature)
            check = self.sign(msg_list[1:5])
            if not compare_digest(signature, check):
                msg = "Invalid Signature: %r" % signature
                raise ValueError(msg)
        if not len(msg_list) >= minlen:
            msg = "malformed message, must have at least %i elements" % minlen
            raise TypeError(msg)
        header = self.unpack(msg_list[1])
        message["header"] = extract_dates(header)
        message["msg_id"] = header["msg_id"]
        message["msg_type"] = header["msg_type"]
        message["parent_header"] = extract_dates(self.unpack(msg_list[2]))
        message["metadata"] = self.unpack(msg_list[3])
        if content:
            message["content"] = self.unpack(msg_list[4])
        else:
            message["content"] = msg_list[4]
        buffers = [memoryview(b) for b in msg_list[5:]]
        if buffers and buffers[0].shape is None:
            # force copy to workaround pyzmq #646
            msg_list = t.cast(t.List[zmq.Message], msg_list)
            buffers = [memoryview(bytes(b.bytes)) for b in msg_list[5:]]
        message["buffers"] = buffers
        if self.debug:
            pprint.pprint(message)  # noqa
        # adapt to the current version
        return adapt(message)

    def unserialize(self, *args: t.Any, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """**DEPRECATED** Use deserialize instead."""
        # pragma: no cover
        warnings.warn(
            "Session.unserialize is deprecated. Use Session.deserialize.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.deserialize(*args, **kwargs)
