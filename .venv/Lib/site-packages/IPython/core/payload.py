# -*- coding: utf-8 -*-
"""Payload system for IPython.

Authors:

* Fernando Perez
* Brian Granger
"""

#-----------------------------------------------------------------------------
#       Copyright (C) 2008-2011 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from traitlets.config.configurable import Configurable
from traitlets import List

#-----------------------------------------------------------------------------
# Main payload class
#-----------------------------------------------------------------------------

class PayloadManager(Configurable):

    _payload = List([])

    def write_payload(self, data, single=True):
        """Include or update the specified `data` payload in the PayloadManager.

        If a previous payload with the same source exists and `single` is True,
        it will be overwritten with the new one.
        """

        if not isinstance(data, dict):
            raise TypeError('Each payload write must be a dict, got: %r' % data)

        if single and 'source' in data:
            source = data['source']
            for i, pl in enumerate(self._payload):
                if 'source' in pl and pl['source'] == source:
                    self._payload[i] = data
                    return

        self._payload.append(data)

    def read_payload(self):
        return self._payload

    def clear_payload(self):
        self._payload = []
