"""A shim module for deprecated imports
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module

from .importstring import import_item


class ShimWarning(Warning):
    """A warning to show when a module has moved, and a shim is in its place."""


class ShimImporter(importlib.abc.MetaPathFinder):
    """Import hook for a shim.

    This ensures that submodule imports return the real target module,
    not a clone that will confuse `is` and `isinstance` checks.
    """
    def __init__(self, src, mirror):
        self.src = src
        self.mirror = mirror

    def _mirror_name(self, fullname):
        """get the name of the mirrored module"""

        return self.mirror + fullname[len(self.src) :]

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.src + "."):
            mirror_name = self._mirror_name(fullname)
            return importlib.util.find_spec(mirror_name)


class ShimModule(types.ModuleType):

    def __init__(self, *args, **kwargs):
        self._mirror = kwargs.pop("mirror")
        src = kwargs.pop("src", None)
        if src:
            kwargs['name'] = src.rsplit('.', 1)[-1]
        super(ShimModule, self).__init__(*args, **kwargs)
        # add import hook for descendent modules
        if src:
            sys.meta_path.append(
                ShimImporter(src=src, mirror=self._mirror)
            )
    
    @property
    def __path__(self):
        return []
    
    @property
    def __spec__(self):
        """Don't produce __spec__ until requested"""
        return import_module(self._mirror).__spec__
    
    def __dir__(self):
        return dir(import_module(self._mirror))
    
    @property
    def __all__(self):
        """Ensure __all__ is always defined"""
        mod = import_module(self._mirror)
        try:
            return mod.__all__
        except AttributeError:
            return [name for name in dir(mod) if not name.startswith('_')]

    def __getattr__(self, key):
        # Use the equivalent of import_item(name), see below
        name = "%s.%s" % (self._mirror, key)
        try:
            return import_item(name)
        except ImportError as e:
            raise AttributeError(key) from e

    def __repr__(self):
        # repr on a module can be called during error handling; make sure
        # it does not fail, even if the import fails
        try:
            return self.__getattr__("__repr__")()
        except AttributeError:
            return f"<ShimModule for {self._mirror!r}>"
