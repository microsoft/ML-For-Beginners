"""Traitlets Python configuration system"""
from __future__ import annotations

import typing as _t

from . import traitlets
from ._version import __version__, version_info
from .traitlets import *
from .utils.bunch import Bunch
from .utils.decorators import signature_has_traits
from .utils.importstring import import_item
from .utils.warnings import warn

__all__ = [
    "traitlets",
    "__version__",
    "version_info",
    "Bunch",
    "signature_has_traits",
    "import_item",
    "Sentinel",
]


class Sentinel(traitlets.Sentinel):  # type:ignore[name-defined, misc]
    def __init__(self, *args: _t.Any, **kwargs: _t.Any) -> None:
        super().__init__(*args, **kwargs)
        warn(
            """
            Sentinel is not a public part of the traitlets API.
            It was published by mistake, and may be removed in the future.
            """,
            DeprecationWarning,
            stacklevel=2,
        )
