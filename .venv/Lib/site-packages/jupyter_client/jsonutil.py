"""Utilities to manipulate JSON objects."""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import math
import numbers
import re
import types
import warnings
from binascii import b2a_base64
from collections.abc import Iterable
from datetime import datetime
from typing import Optional, Union

from dateutil.parser import parse as _dateutil_parse
from dateutil.tz import tzlocal

next_attr_name = "__next__"  # Not sure what downstream library uses this, but left it to be safe

# -----------------------------------------------------------------------------
# Globals and constants
# -----------------------------------------------------------------------------

# timestamp formats
ISO8601 = "%Y-%m-%dT%H:%M:%S.%f"
ISO8601_PAT = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d{1,6})?(Z|([\+\-]\d{2}:?\d{2}))?$"
)

# holy crap, strptime is not threadsafe.
# Calling it once at import seems to help.
datetime.strptime("1", "%d")  # noqa

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------


def _ensure_tzinfo(dt: datetime) -> datetime:
    """Ensure a datetime object has tzinfo

    If no tzinfo is present, add tzlocal
    """
    if not dt.tzinfo:
        # No more naïve datetime objects!
        warnings.warn(
            "Interpreting naive datetime as local %s. Please add timezone info to timestamps." % dt,
            DeprecationWarning,
            stacklevel=4,
        )
        dt = dt.replace(tzinfo=tzlocal())
    return dt


def parse_date(s: Optional[str]) -> Optional[Union[str, datetime]]:
    """parse an ISO8601 date string

    If it is None or not a valid ISO8601 timestamp,
    it will be returned unmodified.
    Otherwise, it will return a datetime object.
    """
    if s is None:
        return s
    m = ISO8601_PAT.match(s)
    if m:
        dt = _dateutil_parse(s)
        return _ensure_tzinfo(dt)
    return s


def extract_dates(obj):
    """extract ISO8601 dates from unpacked JSON"""
    if isinstance(obj, dict):
        new_obj = {}  # don't clobber
        for k, v in obj.items():
            new_obj[k] = extract_dates(v)
        obj = new_obj
    elif isinstance(obj, (list, tuple)):
        obj = [extract_dates(o) for o in obj]
    elif isinstance(obj, str):
        obj = parse_date(obj)
    return obj


def squash_dates(obj):
    """squash datetime objects into ISO8601 strings"""
    if isinstance(obj, dict):
        obj = dict(obj)  # don't clobber
        for k, v in obj.items():
            obj[k] = squash_dates(v)
    elif isinstance(obj, (list, tuple)):
        obj = [squash_dates(o) for o in obj]
    elif isinstance(obj, datetime):
        obj = obj.isoformat()
    return obj


def date_default(obj):
    """DEPRECATED: Use jupyter_client.jsonutil.json_default"""
    warnings.warn(
        "date_default is deprecated since jupyter_client 7.0.0."
        " Use jupyter_client.jsonutil.json_default.",
        stacklevel=2,
    )
    return json_default(obj)


def json_default(obj):
    """default function for packing objects in JSON."""
    if isinstance(obj, datetime):
        obj = _ensure_tzinfo(obj)
        return obj.isoformat().replace('+00:00', 'Z')

    if isinstance(obj, bytes):
        return b2a_base64(obj, newline=False).decode('ascii')

    if isinstance(obj, Iterable):
        return list(obj)

    if isinstance(obj, numbers.Integral):
        return int(obj)

    if isinstance(obj, numbers.Real):
        return float(obj)

    raise TypeError("%r is not JSON serializable" % obj)


# Copy of the old ipykernel's json_clean
# This is temporary, it should be removed when we deprecate support for
# non-valid JSON messages
def json_clean(obj):
    # types that are 'atomic' and ok in json as-is.
    atomic_ok = (str, type(None))

    # containers that we need to convert into lists
    container_to_list = (tuple, set, types.GeneratorType)

    # Since bools are a subtype of Integrals, which are a subtype of Reals,
    # we have to check them in that order.

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, numbers.Integral):
        # cast int to int, in case subclasses override __str__ (e.g. boost enum, #4598)
        return int(obj)

    if isinstance(obj, numbers.Real):
        # cast out-of-range floats to their reprs
        if math.isnan(obj) or math.isinf(obj):
            return repr(obj)
        return float(obj)

    if isinstance(obj, atomic_ok):
        return obj

    if isinstance(obj, bytes):
        # unanmbiguous binary data is base64-encoded
        # (this probably should have happened upstream)
        return b2a_base64(obj, newline=False).decode('ascii')

    if isinstance(obj, container_to_list) or (
        hasattr(obj, '__iter__') and hasattr(obj, next_attr_name)
    ):
        obj = list(obj)

    if isinstance(obj, list):
        return [json_clean(x) for x in obj]

    if isinstance(obj, dict):
        # First, validate that the dict won't lose data in conversion due to
        # key collisions after stringification.  This can happen with keys like
        # True and 'true' or 1 and '1', which collide in JSON.
        nkeys = len(obj)
        nkeys_collapsed = len(set(map(str, obj)))
        if nkeys != nkeys_collapsed:
            msg = (
                'dict cannot be safely converted to JSON: '
                'key collision would lead to dropped values'
            )
            raise ValueError(msg)
        # If all OK, proceed by making the new dict that will be json-safe
        out = {}
        for k, v in obj.items():
            out[str(k)] = json_clean(v)
        return out

    if isinstance(obj, datetime):
        return obj.strftime(ISO8601)

    # we don't understand it, it's probably an unserializable object
    raise ValueError("Can't clean for JSON: %r" % obj)
