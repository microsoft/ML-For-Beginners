"""HTTP cache implementation.
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import BinaryIO, Generator, Optional, Union

from pip._vendor.cachecontrol.cache import SeparateBodyBaseCache
from pip._vendor.cachecontrol.caches import SeparateBodyFileCache
from pip._vendor.requests.models import Response

from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import ensure_dir


def is_from_cache(response: Response) -> bool:
    return getattr(response, "from_cache", False)


@contextmanager
def suppressed_cache_errors() -> Generator[None, None, None]:
    """If we can't access the cache then we can just skip caching and process
    requests as if caching wasn't enabled.
    """
    try:
        yield
    except OSError:
        pass


class SafeFileCache(SeparateBodyBaseCache):
    """
    A file based cache which is safe to use even when the target directory may
    not be accessible or writable.
    """

    def __init__(self, directory: str) -> None:
        assert directory is not None, "Cache directory must not be None."
        super().__init__()
        self.directory = directory

    def _get_cache_path(self, name: str) -> str:
        # From cachecontrol.caches.file_cache.FileCache._fn, brought into our
        # class for backwards-compatibility and to avoid using a non-public
        # method.
        hashed = SeparateBodyFileCache.encode(name)
        parts = list(hashed[:5]) + [hashed]
        return os.path.join(self.directory, *parts)

    def get(self, key: str) -> Optional[bytes]:
        path = self._get_cache_path(key)
        with suppressed_cache_errors():
            with open(path, "rb") as f:
                return f.read()

    def _write(self, path: str, data: bytes) -> None:
        with suppressed_cache_errors():
            ensure_dir(os.path.dirname(path))

            with adjacent_tmp_file(path) as f:
                f.write(data)

            replace(f.name, path)

    def set(
        self, key: str, value: bytes, expires: Union[int, datetime, None] = None
    ) -> None:
        path = self._get_cache_path(key)
        self._write(path, value)

    def delete(self, key: str) -> None:
        path = self._get_cache_path(key)
        with suppressed_cache_errors():
            os.remove(path)
        with suppressed_cache_errors():
            os.remove(path + ".body")

    def get_body(self, key: str) -> Optional[BinaryIO]:
        path = self._get_cache_path(key) + ".body"
        with suppressed_cache_errors():
            return open(path, "rb")

    def set_body(self, key: str, body: bytes) -> None:
        path = self._get_cache_path(key) + ".body"
        self._write(path, body)
