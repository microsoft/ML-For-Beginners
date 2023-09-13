import sys
from typing import Any, Dict

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    # avoid runtime dependency on typing_extensions on py37
    try:
        from typing_extensions import Literal, TypedDict  # type: ignore
    except ImportError:

        class _Literal:
            def __getitem__(self, key):
                return Any

        Literal = _Literal()  # type: ignore

        class TypedDict(Dict):  # type: ignore
            pass
