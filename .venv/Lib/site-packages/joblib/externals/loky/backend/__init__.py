import os
from multiprocessing import synchronize

from .context import get_context


def _make_name():
    return f"/loky-{os.getpid()}-{next(synchronize.SemLock._rand)}"


# monkey patch the name creation for multiprocessing
synchronize.SemLock._make_name = staticmethod(_make_name)

__all__ = ["get_context"]
