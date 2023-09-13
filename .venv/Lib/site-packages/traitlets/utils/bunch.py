"""Yet another implementation of bunch

attribute-access of items on a dict.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.


class Bunch(dict):  # type:ignore[type-arg]
    """A dict with attribute-access"""

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __dir__(self):
        # py2-compat: can't use super because dict doesn't have __dir__
        names = dir({})
        names.extend(self.keys())
        return names
