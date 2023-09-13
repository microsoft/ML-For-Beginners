"""Python implementation of Z85 85-bit encoding

Z85 encoding is a plaintext encoding for a bytestring interpreted as 32bit integers.
Since the chunks are 32bit, a bytestring must be a multiple of 4 bytes.
See ZMQ RFC 32 for details.


"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import struct

# Z85CHARS is the base 85 symbol table
Z85CHARS = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
# Z85MAP maps integers in [0,84] to the appropriate character in Z85CHARS
Z85MAP = {c: idx for idx, c in enumerate(Z85CHARS)}

_85s = [85**i for i in range(5)][::-1]


def encode(rawbytes):
    """encode raw bytes into Z85"""
    # Accepts only byte arrays bounded to 4 bytes
    if len(rawbytes) % 4:
        raise ValueError("length must be multiple of 4, not %i" % len(rawbytes))

    nvalues = len(rawbytes) / 4

    values = struct.unpack('>%dI' % nvalues, rawbytes)
    encoded = []
    for v in values:
        for offset in _85s:
            encoded.append(Z85CHARS[(v // offset) % 85])

    return bytes(encoded)


def decode(z85bytes):
    """decode Z85 bytes to raw bytes, accepts ASCII string"""
    if isinstance(z85bytes, str):
        try:
            z85bytes = z85bytes.encode('ascii')
        except UnicodeEncodeError:
            raise ValueError('string argument should contain only ASCII characters')

    if len(z85bytes) % 5:
        raise ValueError("Z85 length must be multiple of 5, not %i" % len(z85bytes))

    nvalues = len(z85bytes) / 5
    values = []
    for i in range(0, len(z85bytes), 5):
        value = 0
        for j, offset in enumerate(_85s):
            value += Z85MAP[z85bytes[i + j]] * offset
        values.append(value)
    return struct.pack('>%dI' % nvalues, *values)
