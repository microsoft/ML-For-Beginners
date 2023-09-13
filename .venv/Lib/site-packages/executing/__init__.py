"""
Get information about what a frame is currently doing. Typical usage:

    import executing

    node = executing.Source.executing(frame).node
    # node will be an AST node or None
"""

from collections import namedtuple
_VersionInfo = namedtuple('_VersionInfo', ('major', 'minor', 'micro'))
from .executing import Source, Executing, only, NotOneValueFound, cache, future_flags
try:
    from .version import __version__ # type: ignore[import]
    if "dev" in __version__:
        raise ValueError
except Exception:
    # version.py is auto-generated with the git tag when building
    __version__ = "???"
    __version_info__ = _VersionInfo(-1, -1, -1)
else:
    __version_info__ = _VersionInfo(*map(int, __version__.split('.')))


__all__ = ["Source"]
