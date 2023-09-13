"""Declare basic string types unambiguously for various Python versions.

Authors
-------
* MinRK
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import warnings

bytes = bytes
unicode = str
basestring = (str,)


def cast_bytes(s, encoding='utf8', errors='strict'):
    """cast unicode or bytes to bytes"""
    warnings.warn(
        "zmq.utils.strtypes is deprecated in pyzmq 23.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(s, bytes):
        return s
    elif isinstance(s, str):
        return s.encode(encoding, errors)
    else:
        raise TypeError("Expected unicode or bytes, got %r" % s)


def cast_unicode(s, encoding='utf8', errors='strict'):
    """cast bytes or unicode to unicode"""
    warnings.warn(
        "zmq.utils.strtypes is deprecated in pyzmq 23.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, str):
        return s
    else:
        raise TypeError("Expected unicode or bytes, got %r" % s)


# give short 'b' alias for cast_bytes, so that we can use fake b'stuff'
# to simulate b'stuff'
b = asbytes = cast_bytes
u = cast_unicode

__all__ = [
    'asbytes',
    'bytes',
    'unicode',
    'basestring',
    'b',
    'u',
    'cast_bytes',
    'cast_unicode',
]
