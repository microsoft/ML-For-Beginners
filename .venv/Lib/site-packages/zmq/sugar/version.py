"""PyZMQ and 0MQ version functions."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import re
from typing import Match, Tuple, Union, cast

from zmq.backend import zmq_version_info

__version__: str = "25.1.2"
_version_pat = re.compile(r"(\d+)\.(\d+)\.(\d+)(.*)")
_match = cast(Match, _version_pat.match(__version__))
_version_groups = _match.groups()

VERSION_MAJOR = int(_version_groups[0])
VERSION_MINOR = int(_version_groups[1])
VERSION_PATCH = int(_version_groups[2])
VERSION_EXTRA = _version_groups[3].lstrip(".")

version_info: Union[Tuple[int, int, int], Tuple[int, int, int, float]] = (
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
)

if VERSION_EXTRA:
    version_info = (
        VERSION_MAJOR,
        VERSION_MINOR,
        VERSION_PATCH,
        float('inf'),
    )

__revision__: str = ''


def pyzmq_version() -> str:
    """return the version of pyzmq as a string"""
    if __revision__:
        return '+'.join([__version__, __revision__[:6]])
    else:
        return __version__


def pyzmq_version_info() -> Union[Tuple[int, int, int], Tuple[int, int, int, float]]:
    """return the pyzmq version as a tuple of at least three numbers

    If pyzmq is a development version, `inf` will be appended after the third integer.
    """
    return version_info


def zmq_version() -> str:
    """return the version of libzmq as a string"""
    return "%i.%i.%i" % zmq_version_info()


__all__ = [
    'zmq_version',
    'zmq_version_info',
    'pyzmq_version',
    'pyzmq_version_info',
    '__version__',
    '__revision__',
]
