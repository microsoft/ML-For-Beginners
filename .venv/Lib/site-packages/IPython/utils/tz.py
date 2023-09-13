# encoding: utf-8
"""
Timezone utilities

Just UTC-awareness right now
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2013 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from datetime import tzinfo, timedelta, datetime

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------
# constant for zero offset
ZERO = timedelta(0)

class tzUTC(tzinfo):
    """tzinfo object for UTC (zero offset)"""

    def utcoffset(self, d):
        return ZERO

    def dst(self, d):
        return ZERO


UTC = tzUTC()  # type: ignore[abstract]


def utc_aware(unaware):
    """decorator for adding UTC tzinfo to datetime's utcfoo methods"""
    def utc_method(*args, **kwargs):
        dt = unaware(*args, **kwargs)
        return dt.replace(tzinfo=UTC)
    return utc_method

utcfromtimestamp = utc_aware(datetime.utcfromtimestamp)
utcnow = utc_aware(datetime.utcnow)
