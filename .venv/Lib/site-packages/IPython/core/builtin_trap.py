"""
A context manager for managing things injected into :mod:`builtins`.
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
import builtins as builtin_mod

from traitlets.config.configurable import Configurable

from traitlets import Instance


class __BuiltinUndefined(object): pass
BuiltinUndefined = __BuiltinUndefined()

class __HideBuiltin(object): pass
HideBuiltin = __HideBuiltin()


class BuiltinTrap(Configurable):

    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC',
                     allow_none=True)

    def __init__(self, shell=None):
        super(BuiltinTrap, self).__init__(shell=shell, config=None)
        self._orig_builtins = {}
        # We define this to track if a single BuiltinTrap is nested.
        # Only turn off the trap when the outermost call to __exit__ is made.
        self._nested_level = 0
        self.shell = shell
        # builtins we always add - if set to HideBuiltin, they will just
        # be removed instead of being replaced by something else
        self.auto_builtins = {'exit': HideBuiltin,
                              'quit': HideBuiltin,
                              'get_ipython': self.shell.get_ipython,
                              }

    def __enter__(self):
        if self._nested_level == 0:
            self.activate()
        self._nested_level += 1
        # I return self, so callers can use add_builtin in a with clause.
        return self

    def __exit__(self, type, value, traceback):
        if self._nested_level == 1:
            self.deactivate()
        self._nested_level -= 1
        # Returning False will cause exceptions to propagate
        return False

    def add_builtin(self, key, value):
        """Add a builtin and save the original."""
        bdict = builtin_mod.__dict__
        orig = bdict.get(key, BuiltinUndefined)
        if value is HideBuiltin:
            if orig is not BuiltinUndefined: #same as 'key in bdict'
                self._orig_builtins[key] = orig
                del bdict[key]
        else:
            self._orig_builtins[key] = orig
            bdict[key] = value

    def remove_builtin(self, key, orig):
        """Remove an added builtin and re-set the original."""
        if orig is BuiltinUndefined:
            del builtin_mod.__dict__[key]
        else:
            builtin_mod.__dict__[key] = orig

    def activate(self):
        """Store ipython references in the __builtin__ namespace."""

        add_builtin = self.add_builtin
        for name, func in self.auto_builtins.items():
            add_builtin(name, func)

    def deactivate(self):
        """Remove any builtins which might have been added by add_builtins, or
        restore overwritten ones to their previous values."""
        remove_builtin = self.remove_builtin
        for key, val in self._orig_builtins.items():
            remove_builtin(key, val)
        self._orig_builtins.clear()
        self._builtins_added = False
