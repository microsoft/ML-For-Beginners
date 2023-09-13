"""miscellaneous zmq_utils wrapping"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from zmq.error import InterruptedSystemCall, _check_rc, _check_version

from ._cffi import ffi
from ._cffi import lib as C


def has(capability):
    """Check for zmq capability by name (e.g. 'ipc', 'curve')

    .. versionadded:: libzmq-4.1
    .. versionadded:: 14.1
    """
    _check_version((4, 1), 'zmq.has')
    if isinstance(capability, str):
        capability = capability.encode('utf8')
    return bool(C.zmq_has(capability))


def curve_keypair():
    """generate a Z85 key pair for use with zmq.CURVE security

    Requires libzmq (≥ 4.0) to have been built with CURVE support.

    Returns
    -------
    (public, secret) : two bytestrings
        The public and private key pair as 40 byte z85-encoded bytestrings.
    """
    _check_version((3, 2), "curve_keypair")
    public = ffi.new('char[64]')
    private = ffi.new('char[64]')
    rc = C.zmq_curve_keypair(public, private)
    _check_rc(rc)
    return ffi.buffer(public)[:40], ffi.buffer(private)[:40]


def curve_public(private):
    """Compute the public key corresponding to a private key for use
    with zmq.CURVE security

    Requires libzmq (≥ 4.2) to have been built with CURVE support.

    Parameters
    ----------
    private
        The private key as a 40 byte z85-encoded bytestring
    Returns
    -------
    bytestring
        The public key as a 40 byte z85-encoded bytestring.
    """
    if isinstance(private, str):
        private = private.encode('utf8')
    _check_version((4, 2), "curve_public")
    public = ffi.new('char[64]')
    rc = C.zmq_curve_public(public, private)
    _check_rc(rc)
    return ffi.buffer(public)[:40]


def _retry_sys_call(f, *args, **kwargs):
    """make a call, retrying if interrupted with EINTR"""
    while True:
        rc = f(*args)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break


__all__ = ['has', 'curve_keypair', 'curve_public']
