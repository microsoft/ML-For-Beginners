# encoding: utf-8
"""
Timezone utilities

Just UTC-awareness right now

Deprecated since IPython 8.19.0.
"""

# -----------------------------------------------------------------------------
#  Copyright (C) 2013 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import warnings
from datetime import tzinfo, timedelta, datetime

# -----------------------------------------------------------------------------
# Code
# -----------------------------------------------------------------------------
__all__ = ["tzUTC", "utc_aware", "utcfromtimestamp", "utcnow"]


# constant for zero offset
ZERO = timedelta(0)


def __getattr__(name):
    if name not in __all__:
        err = f"IPython.utils.tz is deprecated and has no attribute {name}"
        raise AttributeError(err)

    _warn_deprecated()

    return getattr(name)


def _warn_deprecated():
    msg = "The module `IPython.utils.tz` is deprecated and will be completely removed."
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class tzUTC(tzinfo):
    """tzinfo object for UTC (zero offset)

    Deprecated since IPython 8.19.0.
    """

    _warn_deprecated()

    def utcoffset(self, d):
        return ZERO

    def dst(self, d):
        return ZERO


UTC = tzUTC()  # type: ignore[abstract]


def utc_aware(unaware):
    """decorator for adding UTC tzinfo to datetime's utcfoo methods

    Deprecated since IPython 8.19.0.
    """

    def utc_method(*args, **kwargs):
        _warn_deprecated()
        dt = unaware(*args, **kwargs)
        return dt.replace(tzinfo=UTC)

    return utc_method


utcfromtimestamp = utc_aware(datetime.utcfromtimestamp)
utcnow = utc_aware(datetime.utcnow)
