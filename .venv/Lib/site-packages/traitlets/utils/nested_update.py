# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

from typing import Any, Dict


def nested_update(this: Dict[Any, Any], that: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge two nested dictionaries.

    Effectively a recursive ``dict.update``.

    Examples
    --------
    Merge two flat dictionaries:
    >>> nested_update(
    ...     {'a': 1, 'b': 2},
    ...     {'b': 3, 'c': 4}
    ... )
    {'a': 1, 'b': 3, 'c': 4}

    Merge two nested dictionaries:
    >>> nested_update(
    ...     {'x': {'a': 1, 'b': 2}, 'y': 5, 'z': 6},
    ...     {'x': {'b': 3, 'c': 4}, 'z': 7, '0': 8},
    ... )
    {'x': {'a': 1, 'b': 3, 'c': 4}, 'y': 5, 'z': 7, '0': 8}

    """
    for key, value in this.items():
        if isinstance(value, dict):
            if key in that and isinstance(that[key], dict):
                nested_update(this[key], that[key])
        elif key in that:
            this[key] = that[key]

    for key, value in that.items():
        if key not in this:
            this[key] = value

    return this
