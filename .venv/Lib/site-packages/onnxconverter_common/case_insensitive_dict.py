# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from collections.abc import Mapping, MutableMapping
from collections import OrderedDict


class CaseInsensitiveDict(MutableMapping):
    def __init__(self, data=None, **kwargs):
        self._dict = OrderedDict()
        if data:
            self.update(data, **kwargs)

    def __setitem__(self, key, value):
        self._dict[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._dict[key.lower()][1]

    def __delitem__(self, key):
        del self._dict[key.lower()]

    def __iter__(self):
        return (key for key, _ in self._dict.values())

    def __len__(self):
        return len(self._dict)

    def lower_key_iteritems(self):
        """Like iteritems(), but with lowercase keys."""
        return (
            (lower_key, keyval[1])
            for lower_key, keyval
            in self._dict.items()
        )

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        return dict(self.lower_key_iteritems()) == dict(other.lower_key_iteritems())

    def copy(self):
        return CaseInsensitiveDict(self._dict.values())

    def __repr__(self):
        return str(dict(self.items()))
