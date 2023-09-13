"""JSON serialize to/from utf8 bytes

.. versionchanged:: 22.2
    Remove optional imports of different JSON implementations.
    Now that we require recent Python, unconditionally use the standard library.
    Custom JSON libraries can be used via custom serialization functions.
"""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import json
from typing import Any, Dict, List, Union

# backward-compatibility, unused
jsonmod = json


def dumps(o: Any, **kwargs) -> bytes:
    """Serialize object to JSON bytes (utf-8).

    Keyword arguments are passed along to :py:func:`json.dumps`.
    """
    return json.dumps(o, **kwargs).encode("utf8")


def loads(s: Union[bytes, str], **kwargs) -> Union[Dict, List, str, int, float]:
    """Load object from JSON bytes (utf-8).

    Keyword arguments are passed along to :py:func:`json.loads`.
    """
    if isinstance(s, bytes):
        s = s.decode("utf8")
    return json.loads(s, **kwargs)


__all__ = ['dumps', 'loads']
