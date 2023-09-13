"""0MQ authentication related functions and classes."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import datetime
import glob
import os
from typing import Dict, Optional, Tuple, Union

import zmq

_cert_secret_banner = """#   ****  Generated on {0} by pyzmq  ****
#   ZeroMQ CURVE **Secret** Certificate
#   DO NOT PROVIDE THIS FILE TO OTHER USERS nor change its permissions.

"""


_cert_public_banner = """#   ****  Generated on {0} by pyzmq  ****
#   ZeroMQ CURVE Public Certificate
#   Exchange securely, or use a secure mechanism to verify the contents
#   of this file after exchange. Store public certificates in your home
#   directory, in the .curve subdirectory.

"""


def _write_key_file(
    key_filename: Union[str, os.PathLike],
    banner: str,
    public_key: Union[str, bytes],
    secret_key: Optional[Union[str, bytes]] = None,
    metadata: Optional[Dict[str, str]] = None,
    encoding: str = 'utf-8',
) -> None:
    """Create a certificate file"""
    if isinstance(public_key, bytes):
        public_key = public_key.decode(encoding)
    if isinstance(secret_key, bytes):
        secret_key = secret_key.decode(encoding)
    with open(key_filename, 'w', encoding='utf8') as f:
        f.write(banner.format(datetime.datetime.now()))

        f.write('metadata\n')
        if metadata:
            for k, v in metadata.items():
                if isinstance(k, bytes):
                    k = k.decode(encoding)
                if isinstance(v, bytes):
                    v = v.decode(encoding)
                f.write(f"    {k} = {v}\n")

        f.write('curve\n')
        f.write(f"    public-key = \"{public_key}\"\n")

        if secret_key:
            f.write(f"    secret-key = \"{secret_key}\"\n")


def create_certificates(
    key_dir: Union[str, os.PathLike],
    name: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Create zmq certificates.

    Returns the file paths to the public and secret certificate files.
    """
    public_key, secret_key = zmq.curve_keypair()
    base_filename = os.path.join(key_dir, name)
    secret_key_file = f"{base_filename}.key_secret"
    public_key_file = f"{base_filename}.key"
    now = datetime.datetime.now()

    _write_key_file(public_key_file, _cert_public_banner.format(now), public_key)

    _write_key_file(
        secret_key_file,
        _cert_secret_banner.format(now),
        public_key,
        secret_key=secret_key,
        metadata=metadata,
    )

    return public_key_file, secret_key_file


def load_certificate(
    filename: Union[str, os.PathLike]
) -> Tuple[bytes, Optional[bytes]]:
    """Load public and secret key from a zmq certificate.

    Returns (public_key, secret_key)

    If the certificate file only contains the public key,
    secret_key will be None.

    If there is no public key found in the file, ValueError will be raised.
    """
    public_key = None
    secret_key = None
    if not os.path.exists(filename):
        raise OSError(f"Invalid certificate file: {filename}")

    with open(filename, 'rb') as f:
        for line in f:
            line = line.strip()
            if line.startswith(b'#'):
                continue
            if line.startswith(b'public-key'):
                public_key = line.split(b"=", 1)[1].strip(b' \t\'"')
            if line.startswith(b'secret-key'):
                secret_key = line.split(b"=", 1)[1].strip(b' \t\'"')
            if public_key and secret_key:
                break

    if public_key is None:
        raise ValueError("No public key found in %s" % filename)

    return public_key, secret_key


def load_certificates(directory: Union[str, os.PathLike] = '.') -> Dict[bytes, bool]:
    """Load public keys from all certificates in a directory"""
    certs = {}
    if not os.path.isdir(directory):
        raise OSError(f"Invalid certificate directory: {directory}")
    # Follow czmq pattern of public keys stored in *.key files.
    glob_string = os.path.join(directory, "*.key")

    cert_files = glob.glob(glob_string)
    for cert_file in cert_files:
        public_key, _ = load_certificate(cert_file)
        if public_key:
            certs[public_key] = True
    return certs


__all__ = ['create_certificates', 'load_certificate', 'load_certificates']
