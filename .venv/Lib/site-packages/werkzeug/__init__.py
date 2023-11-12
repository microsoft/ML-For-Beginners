from __future__ import annotations

import typing as t

from .serving import run_simple as run_simple
from .test import Client as Client
from .wrappers import Request as Request
from .wrappers import Response as Response


def __getattr__(name: str) -> t.Any:
    if name == "__version__":
        import importlib.metadata
        import warnings

        warnings.warn(
            "The '__version__' attribute is deprecated and will be removed in"
            " Werkzeug 3.1. Use feature detection or"
            " 'importlib.metadata.version(\"werkzeug\")' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.metadata.version("werkzeug")

    raise AttributeError(name)
