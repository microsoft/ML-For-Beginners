"""serialization utilities for apply messages"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import warnings

warnings.warn(
    "ipykernel.serialize is deprecated. It has moved to ipyparallel.serialize",
    DeprecationWarning,
    stacklevel=2,
)

import pickle
from itertools import chain

try:
    # available since ipyparallel 5.0.0
    from ipyparallel.serialize.canning import (
        CannedObject,
        can,
        can_sequence,
        istype,
        sequence_types,
        uncan,
        uncan_sequence,
    )
    from ipyparallel.serialize.serialize import PICKLE_PROTOCOL
except ImportError:
    # Deprecated since ipykernel 4.3.0
    from ipykernel.pickleutil import (
        PICKLE_PROTOCOL,
        CannedObject,
        can,
        can_sequence,
        istype,
        sequence_types,
        uncan,
        uncan_sequence,
    )

from jupyter_client.session import MAX_BYTES, MAX_ITEMS

# -----------------------------------------------------------------------------
# Serialization Functions
# -----------------------------------------------------------------------------


def _extract_buffers(obj, threshold=MAX_BYTES):
    """extract buffers larger than a certain threshold"""
    buffers = []
    if isinstance(obj, CannedObject) and obj.buffers:
        for i, buf in enumerate(obj.buffers):
            if len(buf) > threshold:
                # buffer larger than threshold, prevent pickling
                obj.buffers[i] = None
                buffers.append(buf)
            # buffer too small for separate send, coerce to bytes
            # because pickling buffer objects just results in broken pointers
            elif isinstance(buf, memoryview):
                obj.buffers[i] = buf.tobytes()
    return buffers


def _restore_buffers(obj, buffers):
    """restore buffers extracted by"""
    if isinstance(obj, CannedObject) and obj.buffers:
        for i, buf in enumerate(obj.buffers):
            if buf is None:
                obj.buffers[i] = buffers.pop(0)


def serialize_object(obj, buffer_threshold=MAX_BYTES, item_threshold=MAX_ITEMS):
    """Serialize an object into a list of sendable buffers.

    Parameters
    ----------
    obj : object
        The object to be serialized
    buffer_threshold : int
        The threshold (in bytes) for pulling out data buffers
        to avoid pickling them.
    item_threshold : int
        The maximum number of items over which canning will iterate.
        Containers (lists, dicts) larger than this will be pickled without
        introspection.

    Returns
    -------
    [bufs] : list of buffers representing the serialized object.
    """
    buffers = []
    if istype(obj, sequence_types) and len(obj) < item_threshold:
        cobj = can_sequence(obj)
        for c in cobj:
            buffers.extend(_extract_buffers(c, buffer_threshold))
    elif istype(obj, dict) and len(obj) < item_threshold:
        cobj = {}
        for k in sorted(obj):
            c = can(obj[k])
            buffers.extend(_extract_buffers(c, buffer_threshold))
            cobj[k] = c
    else:
        cobj = can(obj)
        buffers.extend(_extract_buffers(cobj, buffer_threshold))

    buffers.insert(0, pickle.dumps(cobj, PICKLE_PROTOCOL))
    return buffers


def deserialize_object(buffers, g=None):
    """reconstruct an object serialized by serialize_object from data buffers.

    Parameters
    ----------
    buffers : list of buffers/bytes
    g : globals to be used when uncanning

    Returns
    -------
    (newobj, bufs) : unpacked object, and the list of remaining unused buffers.
    """
    bufs = list(buffers)
    pobj = bufs.pop(0)
    canned = pickle.loads(pobj)  # noqa
    if istype(canned, sequence_types) and len(canned) < MAX_ITEMS:
        for c in canned:
            _restore_buffers(c, bufs)
        newobj = uncan_sequence(canned, g)
    elif istype(canned, dict) and len(canned) < MAX_ITEMS:
        newobj = {}
        for k in sorted(canned):
            c = canned[k]
            _restore_buffers(c, bufs)
            newobj[k] = uncan(c, g)
    else:
        _restore_buffers(canned, bufs)
        newobj = uncan(canned, g)

    return newobj, bufs


def pack_apply_message(f, args, kwargs, buffer_threshold=MAX_BYTES, item_threshold=MAX_ITEMS):
    """pack up a function, args, and kwargs to be sent over the wire

    Each element of args/kwargs will be canned for special treatment,
    but inspection will not go any deeper than that.

    Any object whose data is larger than `threshold`  will not have their data copied
    (only numpy arrays and bytes/buffers support zero-copy)

    Message will be a list of bytes/buffers of the format:

    [ cf, pinfo, <arg_bufs>, <kwarg_bufs> ]

    With length at least two + len(args) + len(kwargs)
    """

    arg_bufs = list(
        chain.from_iterable(serialize_object(arg, buffer_threshold, item_threshold) for arg in args)
    )

    kw_keys = sorted(kwargs.keys())
    kwarg_bufs = list(
        chain.from_iterable(
            serialize_object(kwargs[key], buffer_threshold, item_threshold) for key in kw_keys
        )
    )

    info = dict(nargs=len(args), narg_bufs=len(arg_bufs), kw_keys=kw_keys)

    msg = [pickle.dumps(can(f), PICKLE_PROTOCOL)]
    msg.append(pickle.dumps(info, PICKLE_PROTOCOL))
    msg.extend(arg_bufs)
    msg.extend(kwarg_bufs)

    return msg


def unpack_apply_message(bufs, g=None, copy=True):
    """unpack f,args,kwargs from buffers packed by pack_apply_message()
    Returns: original f,args,kwargs"""
    bufs = list(bufs)  # allow us to pop
    assert len(bufs) >= 2, "not enough buffers!"
    pf = bufs.pop(0)
    f = uncan(pickle.loads(pf), g)  # noqa
    pinfo = bufs.pop(0)
    info = pickle.loads(pinfo)  # noqa
    arg_bufs, kwarg_bufs = bufs[: info["narg_bufs"]], bufs[info["narg_bufs"] :]

    args_list = []
    for _ in range(info["nargs"]):
        arg, arg_bufs = deserialize_object(arg_bufs, g)
        args_list.append(arg)
    args = tuple(args_list)
    assert not arg_bufs, "Shouldn't be any arg bufs left over"

    kwargs = {}
    for key in info["kw_keys"]:
        kwarg, kwarg_bufs = deserialize_object(kwarg_bufs, g)
        kwargs[key] = kwarg
    assert not kwarg_bufs, "Shouldn't be any kwarg bufs left over"

    return f, args, kwargs
