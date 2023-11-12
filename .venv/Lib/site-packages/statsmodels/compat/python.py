"""
Compatibility tools for differences between Python 2 and 3
"""
import sys
from typing import TYPE_CHECKING

PY37 = sys.version_info[:2] == (3, 7)

asunicode = lambda x, _: str(x)  # noqa:E731


__all__ = [
    "asunicode",
    "asstr",
    "asbytes",
    "Literal",
    "lmap",
    "lzip",
    "lrange",
    "lfilter",
    "with_metaclass",
]


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return s.encode("latin1")


def asstr(s):
    if isinstance(s, str):
        return s
    return s.decode("latin1")


# list-producing versions of the major Python iterating functions
def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))


def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs))


def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def lfilter(*args, **kwargs):
    return list(filter(*args, **kwargs))


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

    return type.__new__(metaclass, "temporary_class", (), {})


if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:
    from typing import Any as Literal
